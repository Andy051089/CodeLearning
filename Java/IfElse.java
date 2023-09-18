package hello;
import java.util.Scanner;
public class IfElse {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入一個字母");
		int a = input.nextInt();
		if(a>=65 && a<=90) {
			System.out.println("此字母為大寫"+(char)a);
		}else {
				System.out.println("此字母為小寫"+(char)a);
			}
		}
}
