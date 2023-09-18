package hello;
import java.util.Scanner;
public class Date1 {
	public static void main(String[] args) {
		 Scanner input = new Scanner(System.in); 
		 System.out.print("請輸入天數:");
		 int d = input.nextInt();
		 int ans = d/7;
		 int ans1 = d%7;
		 System.out.println("週數為:"+ans);
		 System.out.println("剩下天數為:"+ans1);	 
	}
}
