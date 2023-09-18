/**
 * 輸入天數後，算出有幾周及剩下天數
 */
package hello;
import java.util.Scanner;
public class date {
	public static void main(String[] args){
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入天數");
		int d = input.nextInt();
		int ans = d/7;
		int ans1 = 7*ans;
		int ans2 = d-ans1;
		System.out.println("週數為:"+ans);
		System.out.println("剩下天數為:"+ans2);
	}
}
/**
package hello;
import java.util.Scanner;
public class date {
	public static void main(String[] args){
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入天數");
		int d = input.nextInt();
		System.out.println("週數為:"+(d/7));
		System.out.println("剩下天數為:"+(d%7));
	}
}
 */